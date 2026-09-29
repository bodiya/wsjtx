! This file is part of WSJT-X.
!
! Copyright (C) 2026 Brian Bodiya, KC1WIH
! License: GPL-3
!
! This program is free software; you can redistribute it and/or modify it under
! the terms of the GNU General Public License as published by the Free Software
! Foundation; either version 3 of the License, or (at your option) any later
! version.
!
! This program is distributed in the hope that it will be useful, but WITHOUT
! ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
! FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
! details.

program test_jtty_back_frame

  ! The slot before a message's first decoded frame (BACK_DEPTH in
  ! jtty_mdecode.f90): frame 1's 13 sync tones are shifted by 2 (mod 4), so
  ! none matches the sync pattern and the blind search cannot find it, while
  ! its 46 payload tones are left intact. Frames 2 and 3 are clean, so the
  ! blind search decodes frame 2 and opens a message with it; nothing before
  ! it can predict frame 1. Only the back try, one frame period before frame
  ! 2 at its frequency, decodes frame 1, and it must go in front of the
  ! message: the update shows the whole message, complete, starting at
  ! frame 1's sync.

  use iso_fortran_env, only: int16
  use jtty_fec, only: is13, TOTAL_K
  use jtty_mdec, only: npending,pending_updates,discard_pending_updates
  use jtty_mod, only: MAX_FRAMES
  implicit none

  integer :: failures

  failures=0
  call run_case('WB9XYZ 599 0123',3,failures)

  if(failures.ne.0) then
     write(*,'(a,i0)') 'test_jtty_back_frame: failures=',failures
     stop 1
  endif
  write(*,'(a)') 'test_jtty_back_frame: all checks passed'

contains

  subroutine run_case(message,nframes,count)
    character(len=*), intent(in) :: message
    integer, intent(in) :: nframes
    integer, intent(inout) :: count
    integer, parameter :: nsps=384
    integer, parameter :: frame_symbols=size(is13)+TOTAL_K
    ! Half a second of silence before frame 1.
    integer, parameter :: npad=6000
    integer :: tones(MAX_FRAMES*frame_symbols)
    integer :: nsym,nsamples,total_samples,i
    integer(int16), allocatable :: pcm(:)
    real, allocatable :: wave(:),combined(:)
    complex, allocatable :: complex_wave(:)
    character(len=80) :: input
    logical :: found
    real :: start

    call discard_pending_updates()
    input=message
    call genjtty(input,tones,nsym)
    call expect(nsym.eq.nframes*frame_symbols, &
         'the message has the expected frame count',count)

    ! Frame 1's sync tones are the first 13 symbols.
    tones(1:size(is13))=mod(is13+2,4)

    ! A frame period of trailing silence so the quarter-step sweep reaches
    ! the last frame (as test_jtty_sticky_retry does).
    nsamples=nsym*nsps
    total_samples=npad+nsamples+frame_symbols*nsps
    allocate(wave(nsamples),complex_wave(total_samples),combined(total_samples))
    allocate(pcm(total_samples))
    call gen_jttywave(tones,nsym,nsps,2.0,12000.0,1500.0, &
         complex_wave,wave,0,nsamples)
    combined=0.
    combined(npad+1:npad+nsamples)=wave
    pcm=int(nint(30000.0*combined),int16)

    ! nfb=1499 keeps channels 1 and 2 (1200-1500, 1500-1800) out, so only
    ! channel 0 and the back try can see the message.
    call rjtty_sub(pcm,1,nsps,200,1499,1500.0,50.0)
    call rjtty_sub(pcm,total_samples,nsps,200,1499,1500.0,50.0)

    found=.false.
    start=-1.0
    do i=1,npending
       if(trim(normalized(pending_updates(i)%decoded)).eq.trim(message) .and. &
            pending_updates(i)%complete) then
          found=.true.
          start=pending_updates(i)%start_tsync
       endif
    enddo
    call expect(found,'the full message decodes, frame 1 in front',count)
    call expect(abs(start-npad/12000.0).lt.0.05, &
         'the message starts at frame 1',count)
    call expect(npending.eq.1,'one message, not two',count)

    deallocate(pcm,wave,complex_wave,combined)
  end subroutine run_case

  function normalized(value) result(result_value)
    character(len=*), intent(in) :: value
    character(len=80) :: result_value
    integer :: i

    result_value=adjustl(value)
    do i=1,len_trim(result_value)
       if(result_value(i:i).eq.'~') result_value(i:i)=' '
    enddo
  end function normalized

  subroutine expect(condition,description,count)
    logical, intent(in) :: condition
    character(len=*), intent(in) :: description
    integer, intent(inout) :: count

    if(.not.condition) then
       count=count+1
       write(*,'(a)') 'FAIL: '//description
    endif
  end subroutine expect

end program test_jtty_back_frame
